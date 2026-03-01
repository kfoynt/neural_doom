class NNEnemyBackendHandler : EventHandler
{
    static const int MAX_SLOTS = 16;
    int fireCooldown[MAX_SLOTS];

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

    private void ApplyEnemySlotCommand(Actor mo, Actor player, int slot)
    {
        int speedPct = ReadSlotInt(slot, "speed", 100);
        int fwdPct = ReadSlotInt(slot, "fwd", 0);
        int sidePct = ReadSlotInt(slot, "side", 0);
        int turnCmd = ReadSlotInt(slot, "turn", 0);
        int aimCmd = ReadSlotInt(slot, "aim", 0);
        int fireCmd = ReadSlotInt(slot, "fire", 0);
        int fireCooldownCmd = ReadSlotInt(slot, "firecd", 12);
        int healthPct = ReadSlotInt(slot, "healthpct", 100);

        if (speedPct < 30)
        {
            speedPct = 30;
        }
        if (speedPct > 250)
        {
            speedPct = 250;
        }
        if (fwdPct < -100) fwdPct = -100;
        if (fwdPct > 100) fwdPct = 100;
        if (sidePct < -100) sidePct = -100;
        if (sidePct > 100) sidePct = 100;
        if (turnCmd < -120) turnCmd = -120;
        if (turnCmd > 120) turnCmd = 120;
        if (fireCooldownCmd < 1) fireCooldownCmd = 1;
        if (fireCooldownCmd > 35) fireCooldownCmd = 35;
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

        mo.Speed = mo.Default.Speed * (double(speedPct) / 100.0);
        mo.target = null;
        mo.threshold = 0;

        mo.Angle += turnCmd;
        if (aimCmd > 0 && player != null && player.health > 0)
        {
            mo.target = player;
            mo.A_FaceTarget();
        }

        double fwd = double(fwdPct) / 100.0;
        double side = double(sidePct) / 100.0;
        double yaw = mo.Angle;
        double c = Cos(yaw);
        double s = Sin(yaw);
        double scale = mo.Speed * 0.45;
        mo.Vel.X = (c * fwd - s * side) * scale;
        mo.Vel.Y = (s * fwd + c * side) * scale;

        if (fireCooldown[slot] > 0)
        {
            fireCooldown[slot]--;
        }
        if (fireCmd > 0 && fireCooldown[slot] <= 0)
        {
            if (player != null && player.health > 0)
            {
                mo.target = player;
            }
            bool launched = mo.SetStateLabel("Missile");
            if (!launched)
            {
                launched = mo.SetStateLabel("Melee");
            }
            if (launched)
            {
                fireCooldown[slot] = fireCooldownCmd;
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
