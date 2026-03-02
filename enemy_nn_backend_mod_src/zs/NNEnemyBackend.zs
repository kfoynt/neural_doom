class NNEnemyBackendHandler : EventHandler
{
    static const int MAX_SLOTS = 16;
    Actor slotActors[MAX_SLOTS];
    int slotCount;

    private double ReadSlotFloat(int slot, String suffix, double fallback)
    {
        String cvarName = String.Format("nn_enemy_cmd_%02d_%s", slot, suffix);
        CVar v = CVar.FindCVar(cvarName);
        if (v == null)
        {
            return fallback;
        }
        return v.GetFloat();
    }

    private double Clamp(double x, double lo, double hi)
    {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    private int ReadSlotLimit()
    {
        int limit = MAX_SLOTS;
        CVar slotVar = CVar.FindCVar("nn_enemy_slot_count");
        if (slotVar != null)
        {
            limit = slotVar.GetInt();
        }
        if (limit < 1) limit = 1;
        if (limit > MAX_SLOTS) limit = MAX_SLOTS;
        return limit;
    }

    private bool ReadKinematicsMode()
    {
        CVar modeVar = CVar.FindCVar("nn_enemy_kinematics_transformer");
        if (modeVar == null)
        {
            return false;
        }
        return modeVar.GetInt() != 0;
    }

    private bool ReadCombatMode()
    {
        CVar modeVar = CVar.FindCVar("nn_enemy_combat_transformer");
        if (modeVar == null)
        {
            return false;
        }
        return modeVar.GetInt() != 0;
    }

    private int DecodeActorCommandId(int commandSlot)
    {
        return int(ReadSlotFloat(commandSlot, "actor_id_raw", -1.0));
    }

    private int FindCommandSlotForActor(int actorId, int slotLimit)
    {
        for (int commandSlot = 0; commandSlot < slotLimit; commandSlot++)
        {
            int commandActorId = DecodeActorCommandId(commandSlot);
            if (commandActorId == actorId)
            {
                return commandSlot;
            }
        }
        return -1;
    }

    private Actor DecodeTargetActor(int commandSlot)
    {
        int targetActorId = int(ReadSlotFloat(commandSlot, "target_actor_id_raw", -1.0));
        if (targetActorId < 0)
        {
            return null;
        }
        if (targetActorId == 0)
        {
            ThinkerIterator pIt = ThinkerIterator.Create("Actor");
            Actor pawn;
            while ((pawn = Actor(pIt.Next())) != null)
            {
                if (pawn.player != null && pawn.health > 0)
                {
                    return pawn;
                }
            }
            return null;
        }
        ThinkerIterator it = ThinkerIterator.Create("Actor");
        Actor target;
        while ((target = Actor(it.Next())) != null)
        {
            if (int(target.id) != targetActorId)
            {
                continue;
            }
            return target;
        }
        return null;
    }

    private void ApplyEnemyCommand(Actor mo, int commandSlot)
    {
        // Hard safety bounds only (raw model command semantics).
        double speedCmd = Clamp(ReadSlotFloat(commandSlot, "speed_norm", 0.0), -64.0, 64.0);
        double fwdCmd = Clamp(ReadSlotFloat(commandSlot, "fwd_norm", 0.0), -256.0, 256.0);
        double sideCmd = Clamp(ReadSlotFloat(commandSlot, "side_norm", 0.0), -256.0, 256.0);
        double turnCmd = Clamp(ReadSlotFloat(commandSlot, "turn_norm", 0.0), -720.0, 720.0);
        double aimCmd = Clamp(ReadSlotFloat(commandSlot, "aim_norm", 0.0), -720.0, 720.0);
        double fireCmd = Clamp(ReadSlotFloat(commandSlot, "fire_norm", 0.0), -256.0, 256.0);
        double fireCdCmd = Clamp(ReadSlotFloat(commandSlot, "firecd_norm", 0.0), -256.0, 256.0);
        double healthCmd = Clamp(ReadSlotFloat(commandSlot, "health_norm", 0.0), -512.0, 512.0);
        double healthRawCmd = Clamp(ReadSlotFloat(commandSlot, "health_raw", 100.0), 0.0, 4096.0);
        double deadRawCmd = Clamp(ReadSlotFloat(commandSlot, "dead_raw", 0.0), 0.0, 1.0);
        double presentCmd = Clamp(ReadSlotFloat(commandSlot, "present_norm", 0.0), -1.0, 1.0);
        double xCmd = Clamp(ReadSlotFloat(commandSlot, "x_raw", mo.Pos.X), -65536.0, 65536.0);
        double yCmd = Clamp(ReadSlotFloat(commandSlot, "y_raw", mo.Pos.Y), -65536.0, 65536.0);
        double angleCmd = Clamp(ReadSlotFloat(commandSlot, "angle_raw", mo.Angle), -8192.0, 8192.0);
        bool kinematicsMode = ReadKinematicsMode();
        bool combatMode = ReadCombatMode();
        _ = fireCdCmd;

        mo.threshold = 0;
        if (combatMode)
        {
            if (deadRawCmd > 0.5)
            {
                mo.health = 0;
                mo.Vel.X = 0.0;
                mo.Vel.Y = 0.0;
                mo.SetStateLabel("Death");
                return;
            }
            double maxHealthWrite = double(mo.Default.health) * 4.0;
            if (maxHealthWrite < 1.0) maxHealthWrite = 1.0;
            mo.health = int(Clamp(healthRawCmd, 1.0, maxHealthWrite));
        }
        if (kinematicsMode && presentCmd > 0.0)
        {
            mo.Angle = angleCmd + aimCmd;
            Vector3 nextPos = (xCmd, yCmd, mo.Pos.Z);
            mo.SetOrigin(nextPos, true);
            mo.Vel.X = 0.0;
            mo.Vel.Y = 0.0;
        }
        else
        {
            mo.Angle += turnCmd + aimCmd;
            double yaw = mo.Angle;
            double c = Cos(yaw);
            double s = Sin(yaw);
            double moveX = c * fwdCmd - s * sideCmd;
            double moveY = s * fwdCmd + c * sideCmd;
            mo.Vel.X = Clamp(moveX * speedCmd, -1024.0, 1024.0);
            mo.Vel.Y = Clamp(moveY * speedCmd, -1024.0, 1024.0);
        }

        Actor targetActor = DecodeTargetActor(commandSlot);
        mo.target = targetActor;

        // Fire pulse is fully decoded upstream in Transformer backend state update.
        // Mod executes only the pulse command and keeps no cadence/timing state.
        double fireDrive = Clamp(fireCmd, -512.0, 512.0);
        if (fireDrive > 0.0)
        {
            mo.SetStateLabel("Missile");
        }

        // Legacy additive health command path (disabled in combat-mode writes).
        if (!combatMode && healthCmd != 0.0)
        {
            double maxHealth = double(mo.Default.health) * 4.0;
            if (maxHealth < 1.0) maxHealth = 1.0;
            mo.health = int(Clamp(double(mo.health) + healthCmd, 1.0, maxHealth));
        }
    }

    override void WorldTick()
    {
        CVar overrideVar = CVar.FindCVar("nn_enemy_override");
        if (overrideVar == null || overrideVar.GetInt() == 0)
        {
            return;
        }

        int slotLimit = ReadSlotLimit();
        ThinkerIterator it = ThinkerIterator.Create("Actor");
        Actor mo;
        slotCount = 0;
        while ((mo = Actor(it.Next())) != null)
        {
            if (!mo.bIsMonster || mo.player != null || mo.health <= 0)
            {
                continue;
            }
            if (slotCount >= slotLimit)
            {
                break;
            }
            slotActors[slotCount] = mo;
            slotCount++;
        }

        for (int i = slotCount; i < MAX_SLOTS; i++)
        {
            slotActors[i] = null;
        }

        // Actor-ID keyed routing: command ownership is independent of runtime slot ordering.
        for (int slot = 0; slot < slotCount; slot++)
        {
            Actor slotActor = slotActors[slot];
            if (slotActor == null || slotActor.health <= 0)
            {
                continue;
            }
            int actorId = int(slotActor.id);
            int commandSlot = FindCommandSlotForActor(actorId, slotLimit);
            if (commandSlot < 0)
            {
                continue;
            }
            ApplyEnemyCommand(slotActor, commandSlot);
        }
    }
}
