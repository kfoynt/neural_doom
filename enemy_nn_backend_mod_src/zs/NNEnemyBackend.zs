class NNEnemyBackendHandler : EventHandler
{
    static const int MAX_SLOTS = 16;

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

    private double Phase01(int salt)
    {
        // Stateless deterministic phase signal for continuous decode.
        int t = (level.time + salt) % 127;
        return double(t) / 126.0;
    }

    private void ApplyEnemySlotCommand(Actor mo, Actor player, int slot)
    {
        double presentNorm = Clamp(ReadSlotFloat(slot, "present_norm", 1.0), -1.0, 1.0);
        double speedNorm = Clamp(ReadSlotFloat(slot, "speed_norm", 0.0), -1.0, 1.0);
        double fwdNorm = Clamp(ReadSlotFloat(slot, "fwd_norm", 0.0), -1.0, 1.0);
        double sideNorm = Clamp(ReadSlotFloat(slot, "side_norm", 0.0), -1.0, 1.0);
        double turnNorm = Clamp(ReadSlotFloat(slot, "turn_norm", 0.0), -1.0, 1.0);
        double aimNorm = Clamp(ReadSlotFloat(slot, "aim_norm", 0.0), -1.0, 1.0);
        double fireNorm = Clamp(ReadSlotFloat(slot, "fire_norm", 0.0), -1.0, 1.0);
        double fireCdNorm = Clamp(ReadSlotFloat(slot, "firecd_norm", 0.5), 0.0, 1.0);
        double healthNorm = Clamp(ReadSlotFloat(slot, "health_norm", 0.5), 0.0, 1.0);
        double targetNorm = Clamp(ReadSlotFloat(slot, "target_norm", 0.0), -1.0, 1.0);

        double speedScale = Clamp(0.30 + 2.20 * (0.5 + 0.5 * speedNorm), 0.30, 2.50);
        double fwd = Clamp(fwdNorm, -1.0, 1.0);
        double side = Clamp(sideNorm, -1.0, 1.0);
        double turnCmd = 120.0 * Clamp(turnNorm, -1.0, 1.0);
        double aimDrive = Clamp(0.5 + 0.5 * aimNorm, 0.0, 1.0);
        double fireDrive = Clamp(0.5 + 0.5 * fireNorm, 0.0, 1.0);
        double cadenceDrive = Clamp(1.0 - fireCdNorm, 0.0, 1.0);
        double targetDrive = Clamp(0.5 + 0.5 * targetNorm, 0.0, 1.0);
        int healthPct = int(20.0 + 280.0 * Clamp(healthNorm, 0.0, 1.0));

        if (healthPct < 20) healthPct = 20;
        if (healthPct > 300) healthPct = 300;

        int targetHealth = int(double(mo.Default.health) * (double(healthPct) / 100.0));
        if (targetHealth < 1)
        {
            targetHealth = 1;
        }
        if (mo.health < targetHealth)
        {
            mo.health += 2;
            if (mo.health > targetHealth)
            {
                mo.health = targetHealth;
            }
        }
        else if (mo.health > targetHealth)
        {
            mo.health -= 2;
            if (mo.health < targetHealth)
            {
                mo.health = targetHealth;
            }
        }

        mo.Speed = mo.Default.Speed * speedScale;
        mo.target = null;
        mo.threshold = 0;

        if (presentNorm < -0.5)
        {
            mo.Vel.X = 0.0;
            mo.Vel.Y = 0.0;
            return;
        }

        mo.Angle += turnCmd;

        bool playerValid = (player != null && player.health > 0);
        if (playerValid)
        {
            mo.target = player;
        }

        // Continuous target policy: no binary target gate, targetDrive controls influence.
        double aimMix = aimDrive * targetDrive;
        if (playerValid && aimMix > Phase01(11 + slot * 13))
        {
            mo.A_FaceTarget();
        }

        double yaw = mo.Angle;
        double c = Cos(yaw);
        double s = Sin(yaw);
        double scale = mo.Speed * 0.45;
        mo.Vel.X = (c * fwd - s * side) * scale;
        mo.Vel.Y = (s * fwd + c * side) * scale;

        // Continuous fire decode: cadence and trigger strength come from model outputs.
        // No rule-based cooldown counter is kept in the mod.
        double fireMix = fireDrive * targetDrive * (0.08 + 0.92 * cadenceDrive);
        if (playerValid && fireMix > Phase01(53 + slot * 17))
        {
            bool launched = mo.SetStateLabel("Missile");
            if (!launched)
            {
                launched = mo.SetStateLabel("Melee");
            }
        }
    }

    override void WorldTick()
    {
        CVar overrideVar = CVar.FindCVar("nn_enemy_override");
        if (overrideVar == null || overrideVar.GetInt() == 0)
        {
            return;
        }

        Actor player = FindAnyPlayer();
        ThinkerIterator it = ThinkerIterator.Create("Actor");
        Actor mo;
        int slot = 0;
        while ((mo = Actor(it.Next())) != null)
        {
            if (!mo.bIsMonster || mo.player != null || mo.health <= 0)
            {
                continue;
            }
            if (slot >= MAX_SLOTS)
            {
                break;
            }
            ApplyEnemySlotCommand(mo, player, slot);
            slot++;
        }
    }
}
