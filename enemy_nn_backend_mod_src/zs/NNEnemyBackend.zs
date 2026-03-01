class NNEnemyBackendHandler : EventHandler
{
    static const int MAX_SLOTS = 16;
    static const double FIRE_TRIGGER_THRESHOLD = 0.10;
    Actor slotActors[MAX_SLOTS];
    int slotCount;

    private Actor FindAnyPlayer()
    {
        ThinkerIterator pit = ThinkerIterator.Create("PlayerPawn");
        Actor pawn = Actor(pit.Next());
        return pawn;
    }

    private int ReadSlotInt(int slot, String suffix, int fallback)
    {
        String cvarName = String.Format("nn_enemy_cmd_%02d_%s", slot, suffix);
        CVar v = CVar.FindCVar(cvarName);
        if (v == null)
        {
            return fallback;
        }
        return v.GetInt();
    }

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

    private Actor DecodeTargetActor(int selfSlot, Actor player, int targetIndex)
    {
        bool playerValid = (player != null && player.health > 0);
        int totalTargets = 1 + slotCount;
        if (totalTargets < 1)
        {
            return playerValid ? player : null;
        }
        if (targetIndex < 0) targetIndex = 0;
        if (targetIndex > totalTargets - 1) targetIndex = totalTargets - 1;

        if (targetIndex == 0)
        {
            return playerValid ? player : null;
        }

        int targetSlot = targetIndex - 1;
        if (targetSlot == selfSlot)
        {
            return playerValid ? player : null;
        }
        if (targetSlot >= 0 && targetSlot < slotCount)
        {
            Actor target = slotActors[targetSlot];
            if (target != null && target.health > 0)
            {
                return target;
            }
        }
        return playerValid ? player : null;
    }

    private void ApplyEnemySlotCommand(Actor mo, int slot, Actor player)
    {
        double presentNorm = Clamp(ReadSlotFloat(slot, "present_norm", 1.0), -1.0, 1.0);
        double speedNorm = Clamp(ReadSlotFloat(slot, "speed_norm", 0.0), 0.0, 3.0);
        double fwdNorm = Clamp(ReadSlotFloat(slot, "fwd_norm", 0.0), -1.0, 1.0);
        double sideNorm = Clamp(ReadSlotFloat(slot, "side_norm", 0.0), -1.0, 1.0);
        double turnNorm = Clamp(ReadSlotFloat(slot, "turn_norm", 0.0), -1.0, 1.0);
        double aimNorm = Clamp(ReadSlotFloat(slot, "aim_norm", 0.0), 0.0, 1.0);
        double fireNorm = Clamp(ReadSlotFloat(slot, "fire_norm", 0.0), 0.0, 1.0);
        double fireCdNorm = Clamp(ReadSlotFloat(slot, "firecd_norm", 0.0), 0.0, 1.0);
        double healthNorm = Clamp(ReadSlotFloat(slot, "health_norm", 1.0), 0.0, 1.0);
        int targetIndex = ReadSlotInt(slot, "targetidx", 0);

        if (presentNorm < -0.5)
        {
            mo.target = null;
            mo.Vel.X = 0.0;
            mo.Vel.Y = 0.0;
            return;
        }

        int targetHealth = int(double(mo.Default.health) * healthNorm);
        if (targetHealth < 1)
        {
            targetHealth = 1;
        }
        mo.health = targetHealth;
        mo.Speed = mo.Default.Speed * speedNorm;
        mo.threshold = 0;

        mo.Angle += 120.0 * turnNorm;
        Actor targetActor = DecodeTargetActor(slot, player, targetIndex);
        mo.target = targetActor;
        if (targetActor != null)
        {
            mo.A_FaceTarget();
        }

        double yaw = mo.Angle;
        double c = Cos(yaw);
        double s = Sin(yaw);
        double scale = mo.Speed * 0.45;
        mo.Vel.X = (c * fwdNorm - s * sideNorm) * scale;
        mo.Vel.Y = (s * fwdNorm + c * sideNorm) * scale;

        // Stateless decode: model emits trigger each tick.
        if (targetActor != null && (fireNorm * fireCdNorm * aimNorm) > FIRE_TRIGGER_THRESHOLD)
        {
            mo.SetStateLabel("Missile");
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
        Actor player = FindAnyPlayer();
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

        for (int slot = 0; slot < slotCount; slot++)
        {
            Actor slotActor = slotActors[slot];
            if (slotActor == null || slotActor.health <= 0)
            {
                continue;
            }
            ApplyEnemySlotCommand(slotActor, slot, player);
        }
    }
}
